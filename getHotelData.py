import pandas as pd
import datetime

monthMapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

debug_rows1 = ['hotel','lead_time','arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies','meal','country','market_segment','distribution_channel','is_repeated_guest','previous_cancellations']

debug_rows2 = ['previous_bookings_not_canceled','reserved_room_type','assigned_room_type','booking_changes','deposit_type','agent','company']

debug_rows3 = ['days_in_waiting_list','customer_type','adr','required_car_parking_spaces']

debug_rows4 = ['total_of_special_requests','reservation_status','reservation_status_date']


def getBalancedByBooleanLabel(df, label):
    df = df.copy()

    yTrue = df[df[label] == 1]
    yFalse = df[df[label] == 0]

    yTrueLen = len(yTrue)
    yFalseLen = len(yFalse)

    targetCount = min(yTrueLen, yFalseLen)

    if (yTrueLen > targetCount):
        yTrue = yTrue.sample(frac = targetCount / yTrueLen, random_state = 200)
    
    if (yFalseLen > targetCount):
        yFalse = yFalse.sample(frac = targetCount / yFalseLen, random_state = 200)
    
    return pd.concat([yTrue, yFalse])

def getHotelData():
    raw_dataset = pd.read_csv('./data/hotel_bookings.csv', na_values = '?', sep = ',')
    df = raw_dataset.copy()

    # df = df[['is_canceled'] + debug_rows4]

    df = df.drop(columns = ['reservation_status','reservation_status_date'])

    # Convert company name & agent to a simpler boolean simply indicating presence
    if 'company' in df.columns:
        df['hasCompany'] = df['company'].notnull()
        df = df.drop(columns = ['company'])

    if 'agent' in df.columns:
        df['hasAgent'] = df['agent'].notnull()
        df = df.drop(columns = ['agent'])

    if 'country' in df.columns:
        df['country'] = df['country'].fillna(value= 'XXX')

    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0)

    if 'arrival_date_month' in df.columns:
        df['arrival_date_month'] = df['arrival_date_month'].map(monthMapping)

    if 'reservation_status_date' in df.columns:
        df['reservation_status_date'] = df['reservation_status_date'].map(
            lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').timestamp()
        )

    # df = getBalancedByBooleanLabel(df, label = 'is_canceled')

    # Train/test and labels split
    (train_data, test_data) = splitDataset(df)

    train_labels = train_data.pop('is_canceled')
    test_labels = test_data.pop('is_canceled')

    return ((train_data, train_labels), (test_data, test_labels))

def splitDataset(df, frac = 0.95):
    df = df.copy()
    train_data = df.sample(frac = frac, random_state = 200)
    test_data = df.drop(train_data.index)

    return (train_data, test_data)

if __name__ == '__main__':
    ((train_data, train_labels), (test_data, test_labels)) = getHotelData()
    
    print('Train size', len(train_data))
    print('Train labels size', len(train_labels))
    print('Test size', len(test_data))
    print('Test labels size', len(test_labels))

